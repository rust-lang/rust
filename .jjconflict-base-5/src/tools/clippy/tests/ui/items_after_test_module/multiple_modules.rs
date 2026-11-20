//@ check-pass

#[cfg(test)]
mod tests {
    #[test]
    fn f() {}
}

#[cfg(test)]
mod more_tests {
    #[test]
    fn g() {}
}
