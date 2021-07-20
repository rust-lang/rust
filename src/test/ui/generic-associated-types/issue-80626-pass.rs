#![feature(generic_associated_types)]

// check-pass

trait Allocator {
    type Allocated<T: ?Sized>;
}

enum LinkedList<A: Allocator> {
    Head,
    Next(u8, A::Allocated<Self>),
}

enum LinkedList2<A: Allocator> {
    Head,
    Next(A::Allocated<Self>, u8),
}

impl Allocator for () {
    type Allocated<T: ?Sized> = Box<T>;
}

fn main() {
    {
        use LinkedList::{Head, Next};
        let mut ll: LinkedList<()> = Next(
            8,
            Box::new(Next(
                0,
                Box::new(Next(
                    6,
                    Box::new(Next(2, Box::new(Next(6, Box::new(Head))))),
                )),
            )),
        );

        while let Next(num, next) = ll {
            println!("{}", num);
            ll = *next;
        }
    }
    {
        use LinkedList2::{Head, Next};
        let mut ll: LinkedList2<()> = Next(
            Box::new(Next(
                Box::new(Next(
                    Box::new(Next(Box::new(Next(Box::new(Head), 6)), 2)),
                    6,
                )),
                0,
            )),
            8,
        );

        while let Next(next, num) = ll {
            println!("{}", num);
            ll = *next;
        }
    }
}
