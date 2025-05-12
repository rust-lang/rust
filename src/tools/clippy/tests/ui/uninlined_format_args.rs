//@aux-build:proc_macros.rs

#![warn(clippy::uninlined_format_args)]
#![allow(named_arguments_used_positionally, unused)]
#![allow(
    clippy::eq_op,
    clippy::format_in_format_args,
    clippy::print_literal,
    clippy::unnecessary_literal_unwrap
)]

extern crate proc_macros;
use proc_macros::with_span;

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
    //~^ uninlined_format_args
    println!("val='{   }'", local_i32); // 3 spaces
    //
    //~^^ uninlined_format_args
    println!("val='{	}'", local_i32); // tab
    //
    //~^^ uninlined_format_args
    println!("val='{ 	}'", local_i32); // space+tab
    //
    //~^^ uninlined_format_args
    println!("val='{	 }'", local_i32); // tab+space
    //
    //~^^ uninlined_format_args
    println!(
        //~^ uninlined_format_args
        "val='{
    }'",
        local_i32
    );
    println!("{}", local_i32);
    //~^ uninlined_format_args
    println!("{}", fn_arg);
    //~^ uninlined_format_args
    println!("{:?}", local_i32);
    //~^ uninlined_format_args
    println!("{:#?}", local_i32);
    //~^ uninlined_format_args
    println!("{:4}", local_i32);
    //~^ uninlined_format_args
    println!("{:04}", local_i32);
    //~^ uninlined_format_args
    println!("{:<3}", local_i32);
    //~^ uninlined_format_args
    println!("{:#010x}", local_i32);
    //~^ uninlined_format_args
    println!("{:.1}", local_f64);
    //~^ uninlined_format_args
    println!("Hello {} is {:.*}", "x", local_i32, local_f64);
    println!("Hello {} is {:.*}", local_i32, 5, local_f64);
    println!("Hello {} is {2:.*}", local_i32, 5, local_f64);
    println!("{} {}", local_i32, local_f64);
    //~^ uninlined_format_args
    println!("{}, {}", local_i32, local_opt.unwrap());
    println!("{}", val);
    //~^ uninlined_format_args
    println!("{}", v = val);
    //~^ uninlined_format_args
    println!("{} {1}", local_i32, 42);
    println!("val='{\t }'", local_i32);
    //~^ uninlined_format_args
    println!("val='{\n }'", local_i32);
    //~^ uninlined_format_args
    println!("val='{local_i32}'", local_i32 = local_i32);
    //~^ uninlined_format_args
    println!("val='{local_i32}'", local_i32 = fn_arg);
    //~^ uninlined_format_args
    println!("{0}", local_i32);
    //~^ uninlined_format_args
    println!("{0:?}", local_i32);
    //~^ uninlined_format_args
    println!("{0:#?}", local_i32);
    //~^ uninlined_format_args
    println!("{0:04}", local_i32);
    //~^ uninlined_format_args
    println!("{0:<3}", local_i32);
    //~^ uninlined_format_args
    println!("{0:#010x}", local_i32);
    //~^ uninlined_format_args
    println!("{0:.1}", local_f64);
    //~^ uninlined_format_args
    println!("{0} {0}", local_i32);
    //~^ uninlined_format_args
    println!("{1} {} {0} {}", local_i32, local_f64);
    //~^ uninlined_format_args
    println!("{0} {1}", local_i32, local_f64);
    //~^ uninlined_format_args
    println!("{1} {0}", local_i32, local_f64);
    //~^ uninlined_format_args
    println!("{1} {0} {1} {0}", local_i32, local_f64);
    //~^ uninlined_format_args
    println!("{1} {0}", "str", local_i32);
    println!("{v}", v = local_i32);
    //~^ uninlined_format_args
    println!("{local_i32:0$}", width);
    //~^ uninlined_format_args
    println!("{local_i32:w$}", w = width);
    //~^ uninlined_format_args
    println!("{local_i32:.0$}", prec);
    //~^ uninlined_format_args
    println!("{local_i32:.p$}", p = prec);
    //~^ uninlined_format_args
    println!("{:0$}", v = val);
    //~^ uninlined_format_args
    println!("{0:0$}", v = val);
    //~^ uninlined_format_args
    println!("{:0$.0$}", v = val);
    //~^ uninlined_format_args
    println!("{0:0$.0$}", v = val);
    //~^ uninlined_format_args
    println!("{0:0$.v$}", v = val);
    //~^ uninlined_format_args
    println!("{0:v$.0$}", v = val);
    //~^ uninlined_format_args
    println!("{v:0$.0$}", v = val);
    //~^ uninlined_format_args
    println!("{v:v$.0$}", v = val);
    //~^ uninlined_format_args
    println!("{v:0$.v$}", v = val);
    //~^ uninlined_format_args
    println!("{v:v$.v$}", v = val);
    //~^ uninlined_format_args
    println!("{:0$}", width);
    //~^ uninlined_format_args
    println!("{:1$}", local_i32, width);
    //~^ uninlined_format_args
    println!("{:w$}", w = width);
    //~^ uninlined_format_args
    println!("{:w$}", local_i32, w = width);
    //~^ uninlined_format_args
    println!("{:.0$}", prec);
    //~^ uninlined_format_args
    println!("{:.1$}", local_i32, prec);
    //~^ uninlined_format_args
    println!("{:.p$}", p = prec);
    //~^ uninlined_format_args
    println!("{:.p$}", local_i32, p = prec);
    //~^ uninlined_format_args
    println!("{:0$.1$}", width, prec);
    //~^ uninlined_format_args
    println!("{:0$.w$}", width, w = prec);
    //~^ uninlined_format_args
    println!("{:1$.2$}", local_f64, width, prec);
    //~^ uninlined_format_args
    println!("{:1$.2$} {0} {1} {2}", local_f64, width, prec);
    //~^ uninlined_format_args
    println!(
        //~^ uninlined_format_args
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
    //~^ uninlined_format_args
    println!("{:w$.p$}", local_i32, w = width, p = prec);
    //~^ uninlined_format_args
    println!("{:w$.p$}", w = width, p = prec);
    //~^ uninlined_format_args
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
        //~^ uninlined_format_args
        "{}",
        // comment with a comma , in it
        val,
    );
    println!("{}", /* comment with a comma , in it */ val);
    //~^ uninlined_format_args

    println!(with_span!("{0} {1}" "{1} {0}"), local_i32, local_f64);
    println!("{}", with_span!(span val));

    if local_i32 > 0 {
        panic!("p1 {}", local_i32);
        //~^ uninlined_format_args
    }
    if local_i32 > 0 {
        panic!("p2 {0}", local_i32);
        //~^ uninlined_format_args
    }
    if local_i32 > 0 {
        panic!("p3 {local_i32}", local_i32 = local_i32);
        //~^ uninlined_format_args
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
    //~^ uninlined_format_args
}

fn _do_not_fire() {
    println!("{:?}", None::<()>);
}

macro_rules! _internal {
    ($($args:tt)*) => {
        println!("{}", format_args!($($args)*))
    };
}

macro_rules! my_println2 {
   ($target:expr, $($args:tt)+) => {{
       if $target {
           _internal!($($args)+)
       }
    }};
}

macro_rules! my_println2_args {
    ($target:expr, $($args:tt)+) => {{
       if $target {
           _internal!("foo: {}", format_args!($($args)+))
       }
    }};
}

macro_rules! my_concat {
    ($fmt:literal $(, $e:expr)*) => {
        println!(concat!("ERROR: ", $fmt), $($e,)*)
    }
}

macro_rules! my_good_macro {
    ($fmt:literal $(, $e:expr)* $(,)?) => {
        println!($fmt $(, $e)*)
    }
}

macro_rules! my_bad_macro {
    ($fmt:literal, $($e:expr),*) => {
        println!($fmt, $($e,)*)
    }
}

macro_rules! my_bad_macro2 {
    ($fmt:literal) => {
        let s = $fmt.clone();
        println!("{}", s);
    };
    ($fmt:literal, $($e:expr)+) => {
        println!($fmt, $($e,)*)
    };
}

// This abomination was suggested by @Alexendoo, may the Rust gods have mercy on their soul...
// https://github.com/rust-lang/rust-clippy/pull/9948#issuecomment-1327965962
macro_rules! used_twice {
    (
        large = $large:literal,
        small = $small:literal,
        $val:expr,
    ) => {
        if $val < 5 {
            println!($small, $val);
        } else {
            println!($large, $val);
        }
    };
}

fn tester2() {
    let local_i32 = 1;
    my_println2_args!(true, "{}", local_i32);
    my_println2!(true, "{}", local_i32);
    my_concat!("{}", local_i32);
    my_good_macro!("{}", local_i32);
    my_good_macro!("{}", local_i32,);
    my_bad_macro!("{}", local_i32);
    my_bad_macro2!("{}", local_i32);
    used_twice! {
        large = "large value: {}",
        small = "small value: {}",
        local_i32,
    };
}

#[clippy::format_args]
macro_rules! usr_println {
    ($target:expr, $($args:tt)*) => {{
        if $target {
            println!($($args)*)
        }
    }};
}

fn user_format() {
    let local_i32 = 1;
    let local_f64 = 2.0;

    usr_println!(true, "val='{}'", local_i32);
    //~^ uninlined_format_args
    usr_println!(true, "{}", local_i32);
    //~^ uninlined_format_args
    usr_println!(true, "{:#010x}", local_i32);
    //~^ uninlined_format_args
    usr_println!(true, "{:.1}", local_f64);
    //~^ uninlined_format_args
}
