A type dependency cycle has been encountered.

Erroneous code example:

```compile_fail,E0391
trait FirstTrait : SecondTrait {

}

trait SecondTrait : FirstTrait {

}
```

The previous example contains a circular dependency between two traits:
`FirstTrait` depends on `SecondTrait` which itself depends on `FirstTrait`.
