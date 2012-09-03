use option::Option;

trait FromStr {
    static fn from_str(s: &str) -> Option<self>;
}

