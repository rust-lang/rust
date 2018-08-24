use m::unexported;
//~^ ERROR: is private

mod m {
    pub fn exported() { }

    fn unexported() { }
}


fn main() { unexported(); }
