


// -*- rust -*-
tag clam<T> { signed(int); unsigned(uint); }

fn getclam<T>() -> clam<T> { ret signed::<T>(42); }

obj impatience<shar T>() {
    fn moreclam() -> clam<T> { be getclam::<T>(); }
}

fn main() { }
