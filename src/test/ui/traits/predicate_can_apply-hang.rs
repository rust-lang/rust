fn f<B>(x: Vec<[[[B; 1]; 1]; 1]>) -> impl PartialEq<B> {
    //~^ ERROR can't compare `Vec<[[[B; 1]; 1]; 1]>` with `B`
    x
}

fn main() {}
