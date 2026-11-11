use super::*;

fn reltim2dur(t: u64) -> Duration {
    Duration::from_micros(t)
}

#[test]
fn test_dur2reltims() {
    assert_eq!(dur2reltims(reltim2dur(0)).collect::<Vec<_>>(), vec![]);
    assert_eq!(dur2reltims(reltim2dur(42)).collect::<Vec<_>>(), vec![42]);
    assert_eq!(
        dur2reltims(reltim2dur(abi::TMAX_RELTIM as u64)).collect::<Vec<_>>(),
        vec![abi::TMAX_RELTIM]
    );
    assert_eq!(
        dur2reltims(reltim2dur(abi::TMAX_RELTIM as u64 + 10000)).collect::<Vec<_>>(),
        vec![abi::TMAX_RELTIM, 10000]
    );
}

#[test]
fn test_dur2tmos() {
    assert_eq!(dur2tmos(reltim2dur(0)).collect::<Vec<_>>(), vec![0]);
    assert_eq!(dur2tmos(reltim2dur(42)).collect::<Vec<_>>(), vec![42]);
    assert_eq!(
        dur2tmos(reltim2dur(abi::TMAX_RELTIM as u64)).collect::<Vec<_>>(),
        vec![abi::TMAX_RELTIM]
    );
    assert_eq!(
        dur2tmos(reltim2dur(abi::TMAX_RELTIM as u64 + 10000)).collect::<Vec<_>>(),
        vec![abi::TMAX_RELTIM, 10000]
    );
}
