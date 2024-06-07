#[unsafe(no_mangle)] //~ ERROR [E0658]
extern "C" fn foo() {

}

fn main() {
    foo();
}
