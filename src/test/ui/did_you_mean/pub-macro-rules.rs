#[macro_use] mod bleh {
    pub macro_rules! foo { //~ ERROR can't qualify macro_rules invocation
        ($n:ident) => (
            fn $n () -> i32 {
                1
            }
        )
    }

}

foo!(meh);

fn main() {
    println!("{}", meh());
}
