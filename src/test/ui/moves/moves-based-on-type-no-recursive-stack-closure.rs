// Tests correct kind-checking of the reason stack closures without the :Copy
// bound must be noncopyable. For details see
// http://smallcultfollowing.com/babysteps/blog/2013/04/30/the-case-of-the-recurring-closure/

struct R<'a> {
    // This struct is needed to create the
    // otherwise infinite type of a fn that
    // accepts itself as argument:
    c: Box<dyn FnMut(&mut R, bool) + 'a>
}

fn innocent_looking_victim() {
    let mut x = Some("hello".to_string());
    conspirator(|f, writer| {
        if writer {
            x = None;
        } else {
            match x {
                Some(ref msg) => {
                    (f.c)(f, true);
                    //~^ ERROR: cannot borrow `*f` as mutable more than once at a time
                    println!("{}", msg);
                },
                None => panic!("oops"),
            }
        }
    })
}

fn conspirator<F>(mut f: F) where F: FnMut(&mut R, bool) {
    let mut r = R {c: Box::new(f)};
    f(&mut r, false) //~ ERROR borrow of moved value
}

fn main() { innocent_looking_victim() }
