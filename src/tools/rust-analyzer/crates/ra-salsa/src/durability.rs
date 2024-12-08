/// Describes how likely a value is to change -- how "durable" it is.
/// By default, inputs have `Durability::LOW` and interned values have
/// `Durability::HIGH`. But inputs can be explicitly set with other
/// durabilities.
///
/// We use durabilities to optimize the work of "revalidating" a query
/// after some input has changed. Ordinarily, in a new revision,
/// queries have to trace all their inputs back to the base inputs to
/// determine if any of those inputs have changed. But if we know that
/// the only changes were to inputs of low durability (the common
/// case), and we know that the query only used inputs of medium
/// durability or higher, then we can skip that enumeration.
///
/// Typically, one assigns low durabilites to inputs that the user is
/// frequently editing. Medium or high durabilities are used for
/// configuration, the source from library crates, or other things
/// that are unlikely to be edited.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Durability(u8);

impl Durability {
    /// Low durability: things that change frequently.
    ///
    /// Example: part of the crate being edited
    pub const LOW: Durability = Durability(0);

    /// Medium durability: things that change sometimes, but rarely.
    ///
    /// Example: a Cargo.toml file
    pub const MEDIUM: Durability = Durability(1);

    /// High durability: things that are not expected to change under
    /// common usage.
    ///
    /// Example: the standard library or something from crates.io
    pub const HIGH: Durability = Durability(2);

    /// The maximum possible durability; equivalent to HIGH but
    /// "conceptually" distinct (i.e., if we add more durability
    /// levels, this could change).
    pub(crate) const MAX: Durability = Self::HIGH;

    /// Number of durability levels.
    pub(crate) const LEN: usize = Self::MAX.index() + 1;

    pub(crate) const fn index(self) -> usize {
        self.0 as usize
    }
}
