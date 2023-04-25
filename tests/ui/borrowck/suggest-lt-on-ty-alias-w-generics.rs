type Lazy<T> = Box<dyn Fn() -> T + 'static>;

fn test(x: &i32) -> Lazy<i32> {
    Box::new(|| {
        //~^ ERROR lifetime may not live long enough
        //~| ERROR closure may outlive the current function
        *x
    })
}

fn main() {}
