/// Whatever to ignore case (`fn` vs `Fn` vs `FN`) or not. Used for recovering.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Case {
    Sensitive,
    Insensitive,
}
