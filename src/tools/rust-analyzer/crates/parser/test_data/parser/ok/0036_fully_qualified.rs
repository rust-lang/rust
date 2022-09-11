// https://github.com/rust-lang/rust-analyzer/issues/311

pub fn foo<S: Iterator>() -> String
where
    <S as Iterator>::Item: Eq,
{
    "".to_owned()
}
