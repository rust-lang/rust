import stackwalk::Word;
import libc::size_t;
import libc::uintptr_t;
import send_map::linear::LinearMap;

export Word;
export gc;
export cleanup_stack_for_failure;

// Mirrors rust_stack.h stk_seg
struct StackSegment {
    let prev: *StackSegment;
    let next: *StackSegment;
    let end: uintptr_t;
    // And other fields which we don't care about...
}

extern mod rustrt {
    fn rust_annihilate_box(ptr: *Word);

    #[rust_stack]
    fn rust_call_tydesc_glue(root: *Word, tydesc: *Word, field: size_t);

    #[rust_stack]
    fn rust_gc_metadata() -> *Word;

    fn rust_get_stack_segment() -> *StackSegment;
}

unsafe fn is_frame_in_segment(fp: *Word, segment: *StackSegment) -> bool {
    let begin: Word = unsafe::reinterpret_cast(&segment);
    let end: Word = unsafe::reinterpret_cast(&(*segment).end);
    let frame: Word = unsafe::reinterpret_cast(&fp);

    return begin <= frame && frame <= end;
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

unsafe fn bump<T, U>(ptr: *T, count: uint) -> *U {
    return unsafe::reinterpret_cast(&ptr::offset(ptr, count));
}

unsafe fn align_to_pointer<T>(ptr: *T) -> *T {
    let align = sys::min_align_of::<*T>();
    let ptr: uint = unsafe::reinterpret_cast(&ptr);
    let ptr = (ptr + (align - 1)) & -align;
    return unsafe::reinterpret_cast(&ptr);
}

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

unsafe fn find_segment_for_frame(fp: *Word, segment: *StackSegment)
    -> {segment: *StackSegment, boundary: bool} {
    // Check if frame is in either current frame or previous frame.
    let in_segment = is_frame_in_segment(fp, segment);
    let in_prev_segment = ptr::is_not_null((*segment).prev) &&
        is_frame_in_segment(fp, (*segment).prev);

    // If frame is not in either segment, walk down segment list until
    // we find the segment containing this frame.
    if !in_segment && !in_prev_segment {
        let mut segment = segment;
        while ptr::is_not_null((*segment).next) &&
            is_frame_in_segment(fp, (*segment).next) {
            segment = (*segment).next;
        }
        return {segment: segment, boundary: false};
    }

    // If frame is in previous frame, then we're at a boundary.
    if !in_segment && in_prev_segment {
        return {segment: (*segment).prev, boundary: true};
    }

    // Otherwise, we're somewhere on the inside of the frame.
    return {segment: segment, boundary: false};
}

unsafe fn walk_gc_roots(mem: Memory, sentinel: **Word, visitor: Visitor) {
    let mut segment = rustrt::rust_get_stack_segment();
    let mut last_ret: *Word = ptr::null();
    // To avoid collecting memory used by the GC itself, skip stack
    // frames until past the root GC stack frame. The root GC stack
    // frame is marked by a sentinel, which is a box pointer stored on
    // the stack.
    let mut reached_sentinel = ptr::is_null(sentinel);
    for stackwalk::walk_stack |frame| {
        unsafe {
            let pc = last_ret;
            let {segment: next_segment, boundary: boundary} =
                find_segment_for_frame(frame.fp, segment);
            segment = next_segment;
            let ret_offset = if boundary { 4 } else { 1 };
            last_ret = *ptr::offset(frame.fp, ret_offset) as *Word;

            if ptr::is_null(pc) {
                again;
            }

            let mut delay_reached_sentinel = reached_sentinel;
            let sp = is_safe_point(pc);
            match sp {
              Some(sp_info) => {
                for walk_safe_point(frame.fp, sp_info) |root, tydesc| {
                    // Skip roots until we see the sentinel.
                    if !reached_sentinel {
                        if root == sentinel {
                            delay_reached_sentinel = true;
                        }
                        again;
                    }

                    // Skip null pointers, which can occur when a
                    // unique pointer has already been freed.
                    if ptr::is_null(*root) {
                        again;
                    }

                    if ptr::is_null(tydesc) {
                        // Root is a generic box.
                        let refcount = **root;
                        if mem | task_local_heap != 0 && refcount != -1 {
                            if !visitor(root, tydesc) { return; }
                        } else if mem | exchange_heap != 0 && refcount == -1 {
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
            reached_sentinel = delay_reached_sentinel;
        }
    }
}

fn gc() {
    unsafe {
        for walk_gc_roots(task_local_heap, ptr::null()) |_root, _tydesc| {
            // FIXME(#2997): Walk roots and mark them.
            io::stdout().write([46]); // .
        }
    }
}

type RootSet = LinearMap<*Word,()>;

fn RootSet() -> RootSet {
    LinearMap()
}

#[cfg(gc)]
fn expect_sentinel() -> bool { true }

#[cfg(nogc)]
fn expect_sentinel() -> bool { false }

// This should only be called from fail, as it will drop the roots
// which are *live* on the stack, rather than dropping those that are
// dead.
fn cleanup_stack_for_failure() {
    unsafe {
        // Leave a sentinel on the stack to mark the current frame. The
        // stack walker will ignore any frames above the sentinel, thus
        // avoiding collecting any memory being used by the stack walker
        // itself.
        //
        // However, when core itself is not compiled with GC, then none of
        // the functions in core will have GC metadata, which means we
        // won't be able to find the sentinel root on the stack. In this
        // case, we can safely skip the sentinel since we won't find our
        // own stack roots on the stack anyway.
        let sentinel_box = ~0;
        let sentinel: **Word = if expect_sentinel() {
            unsafe::reinterpret_cast(&ptr::addr_of(sentinel_box))
        } else {
            ptr::null()
        };

        let mut roots = ~RootSet();
        for walk_gc_roots(need_cleanup, sentinel) |root, tydesc| {
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
