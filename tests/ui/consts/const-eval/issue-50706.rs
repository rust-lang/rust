//@ check-pass

pub struct Stats;

#[derive(PartialEq, Eq)]
pub struct StatVariant {
    pub id: u8,
    _priv: (),
}

#[derive(PartialEq, Eq)]
pub struct Stat {
    pub variant: StatVariant,
    pub index: usize,
    _priv: (),
}

impl Stats {
    pub const TEST: StatVariant = StatVariant{id: 0, _priv: (),};
    #[allow(non_upper_case_globals)]
    pub const A: Stat = Stat{
         variant: Self::TEST,
         index: 0,
         _priv: (),};
}

impl Stat {
    pub fn from_index(variant: StatVariant, index: usize) -> Option<Stat> {
        let stat = Stat{variant, index, _priv: (),};
        match stat {
            Stats::A => Some(Stats::A),
            _ => None,
        }
    }
}

fn main() {}
