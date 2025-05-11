//@ only-cdb
//@ compile-flags:-g

// === CDB TESTS ===================================================================================
// Generic functions cause ambigious breakpoints.
// cdb-command:dx @$debuggerRootNamespace.Debugger.Settings.EngineInitialization.ResolveAmbiguousBreakpoints = true;
// cdb-command:bp `closures.rs:57`
// cdb-command:g
// cdb-command:dx add_closure
// cdb-check:add_closure      [Type: closures::main::closure_env$0]
// cdb-check:     [+0x[...]] _ref__base_value : 0x[...] : 42 [Type: int *]
// cdb-command:dx increment
// cdb-check:increment        [Type: closures::main::closure_env$1]
// cdb-check:     [+0x[...]] _ref__count      : 0x[...] : 2 [Type: int *]
// cdb-command:dx consume_closure
// cdb-check:consume_closure  [Type: closures::main::closure_env$2]
// cdb-check:     [+0x[...]] x                : [...] [Type: alloc::string::String]
// cdb-check:     [+0x[...]] _ref__base_value : 0x[...] : 42 [Type: int *]
// cdb-command:dx simple_closure
// cdb-checksimple_closure   [Type: closures::main::closure_env$5]
// cdb-check:     [+0x[...]] _ref__base_value : 0x[...] : 42 [Type: int *]
// cdb-command:g
// cdb-command:dx first_closure
// cdb-check:first_closure    [Type: closures::main::closure_env$6]
// cdb-check:     [+0x[...]] _ref__variable   : 0x[...] : 1 [Type: int *]
// cdb-check:     [+0x[...]] _ref__constant   : 0x[...] : 2 [Type: int *]
// cdb-check:     [+0x[...]] _ref__a_struct   : 0x[...] [Type: closures::Struct *]
// cdb-check:     [+0x[...]] _ref__struct_ref : 0x[...] [Type: closures::Struct * *]
// cdb-check:     [+0x[...]] _ref__owned_value : 0x[...] [Type: int * *]
// cdb-command:g
// cdb-command:dx many_param_closure
// cdb-check:many_param_closure [Type: closures::main::closure_env$7]
// cdb-check:     [+0x[...]] _ref__base_value : 0x[...] : 42 [Type: int *]
// cdb-command:g
// cdb-command:dv
// cdb-command:dx generic_closure
// cdb-check:generic_closure  [Type: closures::generic_func::closure_env$0<i32>]
// cdb-check:     [+0x[...]] _ref__x          : 0x[...] : 42 [Type: int *]
// cdb-command:g
// cdb-command:dx generic_closure
// cdb-check:generic_closure  [Type: closures::generic_func::closure_env$0<ref$<str$> >]
// cdb-check:     [+0x000] _ref__x          : 0x[...] : "base_value" [Type: ref$<str$> *]
// cdb-command:g
// cdb-command:dx second_closure
// cdb-check:second_closure   [Type: closures::main::closure_env$8]
// cdb-check:     [+0x[...]] _ref__variable   : 0x[...] : 2 [Type: int *]
// cdb-check:     [+0x[...]] _ref__constant   : 0x[...] : 2 [Type: int *]
// cdb-check:     [+0x[...]] _ref__a_struct   : 0x[...] [Type: closures::Struct *]
// cdb-check:     [+0x[...]] _ref__struct_ref : 0x[...] [Type: closures::Struct * *]
// cdb-check:     [+0x[...]] _ref__owned_value : 0x[...] [Type: int * *]

#[inline(never)]
fn generic_func<Tfunc: std::fmt::Debug>(x: Tfunc) {
    let generic_closure = |a: i32| {
        println!("{:?} {}", x, a);
    };

    _zzz(); // #break

    // rustc really wants to inline this closure, so we use black_box instead of calling it
    std::hint::black_box(generic_closure);
}

struct Struct {
    a: isize,
    b: f64,
    c: usize,
}

fn main() {
    let base_value: i32 = 42;
    let mut count: i32 = 0;

    let add_closure = |a: i32, b: i32| a + b + base_value;

    add_closure(40, 2);

    let mut increment = || {
        count += 1;
    };

    increment(); // count: 1
    increment(); // count: 2

    let x = String::from("hello");

    // Define a closure that consumes the captured variable `x`
    let consume_closure = move || {
        drop(x);
        base_value + 1
    };

    consume_closure();

    let paramless_closure = || 42_i32;

    let void_closure = |a: i32| {
        println!("Closure with arg: {:?}", a);
    };

    let simple_closure = || {
        let incremented_value = base_value + 1;
        incremented_value
    };

    let result = /*42; */ add_closure(40, 2);
    println!("Result: {:?}", result);
    void_closure(result);
    let result = simple_closure();
    println!("Result: {:?}", result);

    let mut variable: i32 = 1;
    let constant: i32 = 2;

    let a_struct = Struct { a: -3, b: 4.5, c: 5 };

    _zzz(); // #break

    let struct_ref = &a_struct;
    let owned_value: Box<i32> = Box::new(6);

    {
        let mut first_closure = || {
            variable = constant + a_struct.a as i32 + struct_ref.a as i32 + *owned_value;
        };

        _zzz(); // #break

        first_closure();
    }

    let many_param_closure =
        |a: i32, b: f64, c: usize, d: Struct| base_value + a + b as i32 + c as i32 + d.c as i32;

    _zzz(); // #break

    many_param_closure(1, 2.0, 3, Struct { a: 4, b: 5.0, c: 6 });

    generic_func(42);
    generic_func("base_value");

    {
        let mut second_closure = || {
            variable = constant + a_struct.a as i32 + struct_ref.a as i32 + *owned_value;
        };

        _zzz(); // #break

        second_closure();
    }
}

fn _zzz() {
    ()
}
