// revisions: tail nose
//[tail] run-pass
//[nose] run-fail
#![feature(explicit_tail_calls)]

fn main() {
    with_smol_stack(|| List::from_elem((), 1024 * 32).rec_drop());
}

struct List<T> {
    next: Option<Box<Node<T>>>,
}

struct Node<T> {
    elem: T,
    next: Option<Box<Node<T>>>,
}

impl<T> List<T> {
    fn from_elem(elem: T, n: usize) -> Self
    where
        T: Clone,
    {
        List { next: None }.prepend_n(elem, n)
    }

    fn prepend_n(self, elem: T, n: usize) -> Self
    where
        T: Clone,
    {
        match n {
            0 => self,
            1 => Self { next: p(Node { elem, next: self.next }) },
            _ => {
                #[cfg(tail)]
                become Self { next: p(Node { elem: elem.clone(), next: self.next }) }
                    .prepend_n(elem, n - 1);

                #[cfg(nose)]
                return Self { next: p(Node { elem: elem.clone(), next: self.next }) }
                    .prepend_n(elem, n - 1);
            }
        }
    }

    fn rec_drop(self) {
        if let Some(node) = self.next {
            node.rec_drop()
        }
    }
}

impl<T> Node<T> {
    fn rec_drop(self) {
        if let Some(node) = self.next {
            _ = node.elem;
            become node.rec_drop()
        }
    }
}

fn p<T>(v: T) -> Option<Box<T>> {
    Some(Box::new(v))
}

fn with_smol_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(1024 /* bytes */)
        .name("smol thread".to_owned())
        .spawn(f)
        .unwrap()
        .join()
        .unwrap();
}
