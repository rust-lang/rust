trait hax { } 
impl <A> A: hax { } 

fn perform_hax<T: Owned>(x: @T) -> hax {
    x as hax 
}

fn deadcode() {
    perform_hax(@~"deadcode");
}

fn main() {
    let _ = perform_hax(@42);
}
