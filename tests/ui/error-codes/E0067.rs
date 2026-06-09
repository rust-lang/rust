use std::collections::LinkedList;

fn main() {
    LinkedList::new() += 1; //~ ERROR E0368
                            //~^ ERROR E0067
}
