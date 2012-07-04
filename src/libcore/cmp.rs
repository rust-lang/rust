/// Interfaces used for comparison.

iface ord {
    fn lt(&&other: self) -> bool;
}

iface eq {
    fn eq(&&other: self) -> bool;
}

