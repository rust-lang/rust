//@ edition: 2024
//@ check-pass

// Minimized from reth regression in #161154.

#![forbid(unsafe_code)]
#![allow(dead_code)]

trait TxTuple: 'static {
    type Tx: 'static;
    type Error: 'static;
    fn into_tx(self) -> Self::Tx;
}
trait TxIterator: TxTuple<Tx: Executable<Recovered = Self::Recovered>> {
    type Recovered;
}
trait Executable {
    type Recovered;
}
impl Executable for u8 {
    type Recovered = u8;
}
struct OwnedTx;
impl TxTuple for OwnedTx {
    type Tx = u8;
    type Error = u8;
    fn into_tx(self) -> u8 {
        0
    }
}
impl TxIterator for OwnedTx {
    type Recovered = u8;
}

struct Handle<T, E> {
    tx: Option<T>,
    error: Option<E>,
}
struct Processor;
impl Processor {
    fn tx_iterator_for<'a, T>(&'a self, _: &'a Box<T>) -> Result<impl TxIterator, ()> {
        Ok(OwnedTx)
    }

    fn spawn<T: TxIterator>(
        &mut self,
        txs: T,
    ) -> Result<
        Handle<
            impl Executable<Recovered = T::Recovered> + use<T>,
            impl core::fmt::Debug + 'static + use<T>,
        >,
        (),
    > {
        Ok(Handle { tx: Some(txs.into_tx()), error: Some(0_u8) })
    }

    fn check<T>(&mut self, input: Box<T>) -> Result<(), ()> {
        macro_rules! ensure_ok {
            ($e:expr) => {
                match $e {
                    Ok(value) => value,
                    Err(()) => {
                        drop(input);
                        return Err(());
                    }
                }
            };
        }
        let txs = self.tx_iterator_for(&input)?;
        let handle = ensure_ok!(self.spawn(txs));
        drop(input);
        drop(handle);
        Ok(())
    }
}
fn main() {}
