//@ known-bug: #122989
trait Traitor<const N: N<2> = 1, const N: N<2> = N> {
    fn N(&N) -> N<2> {
        M
    }
}

trait N<const N: Traitor<2> = 12> {}
