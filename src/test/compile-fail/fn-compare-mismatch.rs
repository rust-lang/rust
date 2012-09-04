// xfail-fast
// xfail-test

// This is xfail'd due to bad spurious typecheck error messages.

fn main() {
    fn f() { }
    fn g() { }
    let x = f == g;
    //~^ ERROR mismatched types
    //~^^ ERROR cannot determine a type
}
