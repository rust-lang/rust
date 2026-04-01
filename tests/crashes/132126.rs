//@ known-bug: #132126
trait UnsafeCopy where Self: use<Self> {}
