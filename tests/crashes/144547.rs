//@ known-bug: rust-lang/rust#144547
trait UnderlyingImpl<const MAX_SIZE: usize> {
    type InfoType: LevelInfo;
    type SupportedArray<T>;
}

trait LevelInfo {
    const SUPPORTED_SLOTS: usize;
}

struct Info;

impl LevelInfo for Info {
    const SUPPORTED_SLOTS: usize = 1;
}

struct SomeImpl;

impl<const MAX_SIZE: usize> UnderlyingImpl<MAX_SIZE> for SomeImpl {
    type InfoType = Info;
    // This line makes compiler panic
    type SupportedArray<T> = [T; <Self::InfoType as LevelInfo>::SUPPORTED_SLOTS];
    // But this works
    //type SupportedArray<T> = [T; <Info as LevelInfo>::SUPPORTED_SLOTS];
}
