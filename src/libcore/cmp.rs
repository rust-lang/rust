/// Interfaces used for comparison.

trait ord {
    pure fn lt(&&other: self) -> bool;
}

trait eq {
    pure fn eq(&&other: self) -> bool;
}

