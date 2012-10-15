// xfail-test
struct list<T> {
    element: &self/T,
    mut next: Option<@list<T>>
}

impl<T> list<T>{
    fn addEnd(&self, element: &self/T) {
        let newList = list {
            element: element,
            next: option::None
        };

        self.next = Some(@newList);
    }
}

fn main() {
    let s = @"str";
    let ls: list<@str> = list {
        element: &s,
        next: option::None
    };
    io::println(*ls.element);
}
