struct Test1 {
    #![inline]
    //~^ ERROR an inner attribute is not permitted in this context
}

enum Test2 {
    #![inline]
    //~^ ERROR an inner attribute is not permitted in this context
}

union Test3 {
    //~^ ERROR unions cannot have zero fields
    #![inline]
    //~^ ERROR an inner attribute is not permitted in this context
}

fn main() { }
