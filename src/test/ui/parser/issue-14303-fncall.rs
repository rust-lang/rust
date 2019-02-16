// can't run rustfix because it doesn't handle multipart suggestions correctly
// compile-flags: -Zborrowck=mir
// we need the above to avoid ast borrowck failure in recovered code

struct S<'a, T> {
    a: &'a T,
    b: &'a T,
}

fn foo<'a, 'b>(start: &'a usize, end: &'a usize) {
    let _x = (*start..*end)
        .map(|x| S { a: start, b: end })
        .collect::<Vec<S<_, 'a>>>();
        //~^ ERROR lifetime arguments must be declared prior to type arguments
}

fn main() {}
