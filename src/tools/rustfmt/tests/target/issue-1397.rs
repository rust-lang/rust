pub enum TransactionState {
    Committed(i64),
}

pub enum Packet {
    Transaction { state: TransactionState },
}

fn baz(p: Packet) {
    loop {
        loop {
            loop {
                loop {
                    if let Packet::Transaction {
                        state: TransactionState::Committed(ts, ..),
                        ..
                    } = p
                    {
                        unreachable!()
                    }
                }
            }
        }
    }
}
