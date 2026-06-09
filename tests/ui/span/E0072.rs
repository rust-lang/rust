struct ListNode { //~ ERROR has infinite size
    head: u8,
    tail: Option<ListNode>,
}

fn main() {
}
