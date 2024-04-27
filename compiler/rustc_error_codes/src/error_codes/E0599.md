This error occurs when a method is used on a type which doesn't implement it:

Erroneous code example:

```compile_fail,E0599
struct Mouth;

let x = Mouth;
x.chocolate(); // error: no method named `chocolate` found for type `Mouth`
               //        in the current scope
```

In this case, you need to implement the `chocolate` method to fix the error:

```
struct Mouth;

impl Mouth {
    fn chocolate(&self) { // We implement the `chocolate` method here.
        println!("Hmmm! I love chocolate!");
    }
}

let x = Mouth;
x.chocolate(); // ok!
```
