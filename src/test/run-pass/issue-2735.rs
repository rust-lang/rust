trait hax { } 
impl <A> A: hax { } 

fn perform_hax<T: owned>(x: @T) -> hax {
    x as hax 
}

fn deadcode() {
    perform_hax(@~"deadcode");
}

fn main() {
    perform_hax(@42);
}
