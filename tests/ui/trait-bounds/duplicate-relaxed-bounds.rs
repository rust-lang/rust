fn dupes<T: ?Sized + ?Sized + ?Iterator + ?Iterator>() {}
//~^ ERROR duplicate relaxed `Sized` bounds
//~| ERROR duplicate relaxed `Iterator` bounds
//~| ERROR bound modifier `?` can only be applied to `Sized`
//~| ERROR bound modifier `?` can only be applied to `Sized`

trait Trait {
    // We used to say "type parameter has more than one relaxed default bound"
    // even on *associated types* like here. Test that we no longer do that.
    type Type: ?Sized + ?Sized;
    //~^ ERROR duplicate relaxed `Sized` bounds
    //~| ERROR duplicate relaxed `Sized` bounds
}

// We used to emit an additional error about "multiple relaxed default bounds".
// However, multiple relaxed bounds are actually *fine* if they're distinct.
// Ultimately, we still reject this because `Sized` is
// the only (stable) default trait, so we're fine.
fn not_dupes<T: ?Sized + ?Iterator>() {}
//~^ ERROR bound modifier `?` can only be applied to `Sized`

fn main() {}
