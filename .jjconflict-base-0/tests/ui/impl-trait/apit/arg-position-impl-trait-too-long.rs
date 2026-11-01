struct Header;
struct EntryMetadata;
struct Entry<A, B>(A, B);

trait Tr {
    type EncodedKey;
    type EncodedValue;
}

fn test<C: Tr, R>(
    // This APIT is long, however we shouldn't render the type name with a newline in it.
    y: impl FnOnce(
        &mut Header,
        &mut [EntryMetadata],
        &mut [Entry<C::EncodedKey, C::EncodedValue>]
    ) -> R,
) {
    let () = y;
    //~^ ERROR mismatched types
}

fn main() {}
