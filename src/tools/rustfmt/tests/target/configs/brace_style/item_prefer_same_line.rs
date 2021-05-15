// rustfmt-brace_style: PreferSameLine
// Item brace style

struct Lorem {
    ipsum: bool,
}

struct Dolor<T>
where
    T: Eq, {
    sit: T,
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {}
}
