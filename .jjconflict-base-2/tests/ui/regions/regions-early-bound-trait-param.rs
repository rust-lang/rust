//@ run-pass
// Tests that you can use an early-bound lifetime parameter as
// on of the generic parameters in a trait.

trait Trait<'a> {
    fn long(&'a self) -> isize;
    fn short<'b>(&'b self) -> isize;
}

fn poly_invoke<'c, T: Trait<'c>>(x: &'c T) -> (isize, isize) {
    let l = x.long();
    let s = x.short();
    (l,s)
}

fn object_invoke1<'d>(x: &'d dyn Trait<'d>) -> (isize, isize) {
    let l = x.long();
    let s = x.short();
    (l,s)
}

struct Struct1<'e> {
    f: &'e (dyn Trait<'e>+'e)
}

fn field_invoke1<'f, 'g>(x: &'g Struct1<'f>) -> (isize,isize) {
    let l = x.f.long();
    let s = x.f.short();
    (l,s)
}

struct Struct2<'h, 'i:'h> {
    f: &'h (dyn Trait<'i>+'h)
}

fn object_invoke2<'j, 'k>(x: &'k dyn Trait<'j>) -> isize {
    x.short()
}

fn field_invoke2<'l, 'm, 'n>(x: &'n Struct2<'l,'m>) -> isize {
    x.f.short()
}

trait MakerTrait {
    fn mk() -> Self;
}

fn make_val<T:MakerTrait>() -> T {
    MakerTrait::mk()
}

trait RefMakerTrait<'q> {
    fn mk(_: Self) -> &'q Self;
}

fn make_ref<'r, T:RefMakerTrait<'r>>(t:T) -> &'r T {
    RefMakerTrait::mk(t)
}

impl<'s> Trait<'s> for (isize,isize) {
    fn long(&'s self) -> isize {
        let &(x,_) = self;
        x
    }
    fn short<'b>(&'b self) -> isize {
        let &(_,y) = self;
        y
    }
}

impl<'t> MakerTrait for Box<dyn Trait<'t>+'static> {
    fn mk() -> Box<dyn Trait<'t>+'static> {
        let tup: Box<(isize, isize)> = Box::new((4,5));
        tup as Box<dyn Trait>
    }
}

enum List<'l> {
    Cons(isize, &'l List<'l>),
    Null
}

impl<'l> List<'l> {
    fn car<'m>(&'m self) -> isize {
        match self {
            &List::Cons(car, _) => car,
            &List::Null => panic!(),
        }
    }
    fn cdr<'n>(&'n self) -> &'l List<'l> {
        match self {
            &List::Cons(_, cdr) => cdr,
            &List::Null => panic!(),
        }
    }
}

impl<'t> RefMakerTrait<'t> for List<'t> {
    fn mk(l:List<'t>) -> &'t List<'t> {
        l.cdr()
    }
}

pub fn main() {
    let t = (2,3);
    let o = &t as &dyn Trait;
    let s1 = Struct1 { f: o };
    let s2 = Struct2 { f: o };
    assert_eq!(poly_invoke(&t), (2,3));
    assert_eq!(object_invoke1(&t), (2,3));
    assert_eq!(field_invoke1(&s1), (2,3));
    assert_eq!(object_invoke2(&t), 3);
    assert_eq!(field_invoke2(&s2), 3);

    let m : Box<dyn Trait> = make_val();
    // assert_eq!(object_invoke1(&*m), (4,5));
    //            ~~~~~~~~~~~~~~~~~~~
    // this call yields a compilation error; see ui/span/dropck-object-cycle.rs
    // for details.
    assert_eq!(object_invoke2(&*m), 5);

    // The RefMakerTrait above is pretty strange (i.e., it is strange
    // to consume a value of type T and return a &T).  Easiest thing
    // that came to my mind: consume a cell of a linked list and
    // return a reference to the list it points to.
    let l0 = List::Null;
    let l1 = List::Cons(1, &l0);
    let l2 = List::Cons(2, &l1);
    let rl1 = &l1;
    let r  = make_ref(l2);
    assert_eq!(rl1.car(), r.car());
}
