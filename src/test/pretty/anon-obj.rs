// pp-exact

fn main() {
    let my_obj =
        obj () {
            fn foo() { }
        };
    let my_ext_obj =
        obj () {
            fn foo() { }
            with
            my_obj
        };
}
