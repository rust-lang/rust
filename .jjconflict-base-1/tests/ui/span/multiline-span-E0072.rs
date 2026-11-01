// It should just use the entire body instead of pointing at the next two lines
struct //~ ERROR has infinite size
ListNode
{
    head: u8,
    tail: Option<ListNode>,
}

fn main() {
}
