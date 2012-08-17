import stackwalk::Word;
import libc::size_t;
import send_map::linear::LinearMap;

extern mod rustrt {
    fn rust_annihilate_box(ptr: *Word);

    #[rust_stack]
    fn rust_gc_metadata() -> *Word;

    #[rust_stack]
    fn rust_call_tydesc_glue(root: *Word, tydesc: *Word, field: size_t);
}

type SafePoint = { sp_meta: *Word, fn_meta: *Word };

unsafe fn is_safe_point(pc: *Word) -> Option<SafePoint> {
    let module_meta = rustrt::rust_gc_metadata();
    let num_safe_points_ptr: *u32 = unsafe::reinterpret_cast(&module_meta);
    let num_safe_points = *num_safe_points_ptr as Word;
    let safe_points: *Word =
        ptr::offset(unsafe::reinterpret_cast(&module_meta), 1);

    if ptr::is_null(pc) {
        return None;
    }

    let mut sp = 0 as Word;
    while sp < num_safe_points {
        let sp_loc = *ptr::offset(safe_points, sp*3) as *Word;
        if sp_loc == pc {
            return Some(
                {sp_meta: *ptr::offset(safe_points, sp*3 + 1) as *Word,
                 fn_meta: *ptr::offset(safe_points, sp*3 + 2) as *Word});
        }
        sp += 1;
    }
    return None;
}

type Visitor = fn(root: **Word, tydesc: *Word) -> bool;

unsafe fn walk_safe_point(fp: *Word, sp: SafePoint, visitor: Visitor) {
    let fp_bytes: *u8 = unsafe::reinterpret_cast(&fp);
    let sp_meta_u32s: *u32 = unsafe::reinterpret_cast(&sp.sp_meta);

    let num_stack_roots = *sp_meta_u32s as uint;
    let num_reg_roots = *ptr::offset(sp_meta_u32s, 1) as uint;

    let stack_roots: *u32 =
        unsafe::reinterpret_cast(&ptr::offset(sp_meta_u32s, 2));
    let reg_roots: *u8 =
        unsafe::reinterpret_cast(&ptr::offset(stack_roots, num_stack_roots));
    let addrspaces: *Word =
        unsafe::reinterpret_cast(&ptr::offset(reg_roots, num_reg_roots));
    let tydescs: ***Word =
        unsafe::reinterpret_cast(&ptr::offset(addrspaces, num_stack_roots));

    // Stack roots
    let mut sri = 0;
    while sri < num_stack_roots {
        if *ptr::offset(addrspaces, sri) >= 1 {
            let root =
                ptr::offset(fp_bytes, *ptr::offset(stack_roots, sri) as Word)
                as **Word;
            let tydescpp = ptr::offset(tydescs, sri);
            let tydesc = if ptr::is_not_null(tydescpp) &&
                ptr::is_not_null(*tydescpp) {
                **tydescpp
            } else {
                ptr::null()
            };
            if !visitor(root, tydesc) { return; }
        }
        sri += 1;
    }

    // Register roots
    let mut rri = 0;
    while rri < num_reg_roots {
        if *ptr::offset(addrspaces, num_stack_roots + rri) == 1 {
            // FIXME(#2997): Need to find callee saved registers on the stack.
        }
        rri += 1;
    }
}

type Memory = uint;

const task_local_heap: Memory = 1;
const exchange_heap:   Memory = 2;
const stack:           Memory = 4;

const need_cleanup:    Memory = exchange_heap | stack;

unsafe fn walk_gc_roots(mem: Memory, visitor: Visitor) {
    let mut last_ret: *Word = ptr::null();
    for stackwalk::walk_stack |frame| {
        unsafe {
            if ptr::is_not_null(last_ret) {
                let sp = is_safe_point(last_ret);
                match sp {
                  Some(sp_info) => {
                    for walk_safe_point(frame.fp, sp_info) |root, tydesc| {
                        if ptr::is_null(tydesc) {
                            // Root is a generic box.
                            let refcount = **root;
                            if mem | task_local_heap != 0 && refcount != -1 {
                                if !visitor(root, tydesc) { return; }
                            } else if mem | exchange_heap != 0 {
                                if !visitor(root, tydesc) { return; }
                            }
                        } else {
                            // Root is a non-immediate.
                            if mem | stack != 0 {
                                if !visitor(root, tydesc) { return; }
                            }
                        }
                    }
                  }
                  None => ()
                }
            }
            last_ret = *ptr::offset(frame.fp, 1) as *Word;
        }
    }
}

fn gc() {
    unsafe {
        for walk_gc_roots(task_local_heap) |_root, _tydesc| {
            // FIXME(#2997): Walk roots and mark them.
            io::stdout().write([46]); // .
        }
    }
}

type RootSet = LinearMap<*Word,()>;

fn RootSet() -> RootSet {
    LinearMap()
}

// This should only be called from fail, as it will drop the roots
// which are *live* on the stack, rather than dropping those that are
// dead.
fn cleanup_stack_for_failure() {
    unsafe {
        let mut roots = ~RootSet();
        for walk_gc_roots(need_cleanup) |root, tydesc| {
            // Track roots to avoid double frees.
            if option::is_some(roots.find(&*root)) {
                again;
            }
            roots.insert(*root, ());

            if ptr::is_null(tydesc) {
                rustrt::rust_annihilate_box(*root);
            } else {
                rustrt::rust_call_tydesc_glue(*root, tydesc, 3 as size_t);
            }
        }
    }
}
