//@ edition: 2021

async fn test<T>(_: &u8) {
    let _: &'static T;
    //~^ ERROR the parameter type `T` may not live long enough
}

fn main() {}
