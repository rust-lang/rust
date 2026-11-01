//@error-in-other-file:
#[path = "auxiliary/submodule.rs"]
mod submodule;

#[cfg(test)]
mod tests {
    #[test]
    fn t() {}
}
