trait MyTrait {
    #[doc = MyTrait]
    //~^ ERROR attribute value must be a literal
    fn myfun();
}

fn main() {}
