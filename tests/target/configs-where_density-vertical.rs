// rustfmt-where_density: Vertical
// Where density

trait Lorem {
    fn ipsum<Dolor>(dolor: Dolor) -> Sit
        where Dolor: Eq;

    fn ipsum<Dolor>(dolor: Dolor) -> Sit
        where Dolor: Eq
    {
        // body
    }
}
