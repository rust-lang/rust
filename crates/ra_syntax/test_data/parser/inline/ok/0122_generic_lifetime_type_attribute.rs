fn foo<#[derive(Lifetime)] 'a, #[derive(Type)] T>(_: &'a T) {
}
