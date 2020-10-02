Hello, welcome to `std::simd`!

It seems this pull request template checklist was created while a lot of vector math ops were being implemented, and only really applies to ops. Feel free to delete everything here if it's not applicable, or ask for help if you're not sure what it means!

For a given vector math operation on TxN, please add tests for interactions with:
  - [ ] `T::MAX`
  - [ ] `T::MIN`
  - [ ] -1
  - [ ] 1
  - [ ] 0


For a given vector math operation on TxN where T is a float, please add tests for test interactions with:
  - [ ] a really large number, larger than the mantissa
  - [ ] a really small "subnormal" number
  - [ ] NaN
  - [ ] Infinity
  - [ ] Negative Infinity
