//@ compile-flags: --diagnostic-width=145

fn main() {
    // ignore-tidy-linelength
    let _: () = (); let _: () = (); let _: () = (); let _: () = (); let _: () = (); let _: () = (); let _: () = (); let _: () = (); let _: () = (); let _: () = (); let _: () = (); let _: () = (); let _: () = (); let _: () = (); let _: () = 42; let _: () = (); let _: () = (); let _: () = (); let _: () = (); let _: () = (); let _: () = (); let _: () = (); let _: () = ();
//~^ ERROR mismatched types
}
