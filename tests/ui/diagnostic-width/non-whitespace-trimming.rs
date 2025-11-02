//@ compile-flags: --diagnostic-width=145
// ignore-tidy-linelength

fn main() {
    let _: () = (); let _: () = (); let _: () = (); let _: () = (); let _: () = (); let _: () = (); let _: () = (); let _: () = (); let _: () = (); let _: () = (); let _: () = (); let _: () = (); let _: () = (); let _: () = (); let _: () = 42; let _: () = (); let _: () = (); let _: () = (); let _: () = (); let _: () = (); let _: () = (); let _: () = (); let _: () = ();
//~^ ERROR mismatched types
}
