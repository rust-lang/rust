// aux-build:proc_macro_with_span.rs
// run-rustfix
#![warn(clippy::uninlined_format_args)]
#![allow(named_arguments_used_positionally, unused_imports, unused_macros, unused_variables)]
#![allow(clippy::eq_op, clippy::format_in_format_args, clippy::print_literal)]

extern crate proc_macro_with_span;
use proc_macro_with_span::with_span;

macro_rules! no_param_str {
    () => {
        "{}"
    };
}

macro_rules! my_println {
   ($($args:tt),*) => {{
        println!($($args),*)
    }};
}

macro_rules! my_println_args {
    ($($args:tt),*) => {{
        println!("foo: {}", format_args!($($args),*))
    }};
}

fn tester(fn_arg: i32) {
    let local_i32 = 1;
    let local_f64 = 2.0;
    let local_opt: Option<i32> = Some(3);
    let width = 4;
    let prec = 5;
    let val = 6;

    // make sure this file hasn't been corrupted with tabs converted to spaces
    // let _ = '	';  // <- this is a single tab character
    let _: &[u8; 3] = b"	 	"; // <- <tab><space><tab>

    println!("val='{}'", local_i32);
    println!("val='{   }'", local_i32); // 3 spaces
    println!("val='{	}'", local_i32); // tab
    println!("val='{ 	}'", local_i32); // space+tab
    println!("val='{	 }'", local_i32); // tab+space
    println!(
        "val='{
    }'",
        local_i32
    );
    println!("{}", local_i32);
    println!("{}", fn_arg);
    println!("{:?}", local_i32);
    println!("{:#?}", local_i32);
    println!("{:4}", local_i32);
    println!("{:04}", local_i32);
    println!("{:<3}", local_i32);
    println!("{:#010x}", local_i32);
    println!("{:.1}", local_f64);
    println!("Hello {} is {:.*}", "x", local_i32, local_f64);
    println!("Hello {} is {:.*}", local_i32, 5, local_f64);
    println!("Hello {} is {2:.*}", local_i32, 5, local_f64);
    println!("{} {}", local_i32, local_f64);
    println!("{}, {}", local_i32, local_opt.unwrap());
    println!("{}", val);
    println!("{}", v = val);
    println!("{} {1}", local_i32, 42);
    println!("val='{\t }'", local_i32);
    println!("val='{\n }'", local_i32);
    println!("val='{local_i32}'", local_i32 = local_i32);
    println!("val='{local_i32}'", local_i32 = fn_arg);
    println!("{0}", local_i32);
    println!("{0:?}", local_i32);
    println!("{0:#?}", local_i32);
    println!("{0:04}", local_i32);
    println!("{0:<3}", local_i32);
    println!("{0:#010x}", local_i32);
    println!("{0:.1}", local_f64);
    println!("{0} {0}", local_i32);
    println!("{1} {} {0} {}", local_i32, local_f64);
    println!("{0} {1}", local_i32, local_f64);
    println!("{1} {0}", local_i32, local_f64);
    println!("{1} {0} {1} {0}", local_i32, local_f64);
    println!("{1} {0}", "str", local_i32);
    println!("{v}", v = local_i32);
    println!("{local_i32:0$}", width);
    println!("{local_i32:w$}", w = width);
    println!("{local_i32:.0$}", prec);
    println!("{local_i32:.p$}", p = prec);
    println!("{:0$}", v = val);
    println!("{0:0$}", v = val);
    println!("{:0$.0$}", v = val);
    println!("{0:0$.0$}", v = val);
    println!("{0:0$.v$}", v = val);
    println!("{0:v$.0$}", v = val);
    println!("{v:0$.0$}", v = val);
    println!("{v:v$.0$}", v = val);
    println!("{v:0$.v$}", v = val);
    println!("{v:v$.v$}", v = val);
    println!("{:0$}", width);
    println!("{:1$}", local_i32, width);
    println!("{:w$}", w = width);
    println!("{:w$}", local_i32, w = width);
    println!("{:.0$}", prec);
    println!("{:.1$}", local_i32, prec);
    println!("{:.p$}", p = prec);
    println!("{:.p$}", local_i32, p = prec);
    println!("{:0$.1$}", width, prec);
    println!("{:0$.w$}", width, w = prec);
    println!("{:1$.2$}", local_f64, width, prec);
    println!("{:1$.2$} {0} {1} {2}", local_f64, width, prec);
    println!(
        "{0:1$.2$} {0:2$.1$} {1:0$.2$} {1:2$.0$} {2:0$.1$} {2:1$.0$}",
        local_i32, width, prec,
    );
    println!(
        "{0:1$.2$} {0:2$.1$} {1:0$.2$} {1:2$.0$} {2:0$.1$} {2:1$.0$} {3}",
        local_i32,
        width,
        prec,
        1 + 2
    );
    println!("Width = {}, value with width = {:0$}", local_i32, local_f64);
    println!("{:w$.p$}", local_i32, w = width, p = prec);
    println!("{:w$.p$}", w = width, p = prec);
    println!("{}", format!("{}", local_i32));
    my_println!("{}", local_i32);
    my_println_args!("{}", local_i32);

    // these should NOT be modified by the lint
    println!(concat!("nope ", "{}"), local_i32);
    println!("val='{local_i32}'");
    println!("val='{local_i32 }'");
    println!("val='{local_i32	}'"); // with tab
    println!("val='{local_i32\n}'");
    println!("{}", usize::MAX);
    println!("{}", local_opt.unwrap());
    println!(
        "val='{local_i32
    }'"
    );
    println!(no_param_str!(), local_i32);

    println!(
        "{}",
        // comment with a comma , in it
        val,
    );
    println!("{}", /* comment with a comma , in it */ val);

    println!(with_span!("{0} {1}" "{1} {0}"), local_i32, local_f64);
    println!("{}", with_span!(span val));

    if local_i32 > 0 {
        panic!("p1 {}", local_i32);
    }
    if local_i32 > 0 {
        panic!("p2 {0}", local_i32);
    }
    if local_i32 > 0 {
        panic!("p3 {local_i32}", local_i32 = local_i32);
    }
    if local_i32 > 0 {
        panic!("p4 {local_i32}");
    }
}

fn main() {
    tester(42);
}

#[clippy::msrv = "1.57"]
fn _under_msrv() {
    let local_i32 = 1;
    println!("don't expand='{}'", local_i32);
}

#[clippy::msrv = "1.58"]
fn _meets_msrv() {
    let local_i32 = 1;
    println!("expand='{}'", local_i32);
}

fn _do_not_fire() {
    println!("{:?}", None::<()>);
}
