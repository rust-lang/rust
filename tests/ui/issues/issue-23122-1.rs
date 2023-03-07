trait Next {
    type Next: Next;
}

struct GetNext<T: Next> {
    t: T,
}

impl<T: Next> Next for GetNext<T> {
    type Next = <GetNext<T> as Next>::Next;
    //~^ ERROR overflow evaluating the requirement
}

fn main() {}
