#![feature(macro_metavar_expr_concat)]
#![crate_type = "lib"]

macro_rules! make_stuff {
    ($prefix:literal, $name:ident) => {
        #[deprecated(since = "1.0.0", note = ${concat_str($prefix, " ", $name, " trait")})]
        #[diagnostic::on_unimplemented(message = ${concat_str("please do not the ", $name)})]
        pub trait ${concat(New, $name)} {}
    }
}

make_stuff!("blah blah blah", AAAAAA);

pub fn foo(x: impl NewAAAAAA) {}
//~^ WARN use of deprecated trait `NewAAAAAA`: blah blah blah AAAAAA trait [deprecated]

fn bar(){
    foo(());
    //~^ ERROR please do not the AAAAAA [E0277]
}
