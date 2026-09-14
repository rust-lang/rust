// Based on this: https://rust-lang.zulipchat.com/#narrow/channel/326866-t-types.2Fnominated/topic/.23157407.3A.201.2E97.20beta.20regression.3A.20.22conflicting.20implementations.E2.80.A6/near/601113926
pub trait Row {
    type Database;
}

pub struct PgRow;

impl Row for PgRow {
    type Database = ();
}
