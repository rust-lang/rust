iface hax { } 
impl <A> of hax for A { } 

fn perform_hax<T>(x: @T) -> hax {
    x as hax 
}

fn deadcode() {
    perform_hax(@"deadcode");
}

fn main() {
    perform_hax(@42);
}
