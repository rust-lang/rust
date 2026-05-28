# Changelog

## 1.1.0

* add `TextRange::ordering` method

## 1.0.0 :tada:

* the carate is renamed to `text-size` from `text_unit`

Transition table:
- `TextUnit::of_char(c)`                ⟹ `TextSize::of(c)`
- `TextUnit::of_str(s)`                 ⟹ `TextSize::of(s)`
- `TextUnit::from_usize(size)`          ⟹ `TextSize::try_from(size).unwrap_or_else(|| panic!(_))`
- `unit.to_usize()`                     ⟹ `usize::from(size)`
- `TextRange::from_to(from, to)`        ⟹ `TextRange::new(from, to)`
- `TextRange::offset_len(offset, size)` ⟹ `TextRange::from_len(offset, size)`
- `range.start()`                       ⟹ `range.start()`
- `range.end()`                         ⟹ `range.end()`
- `range.len()`                         ⟹ `range.len()`
- `range.is_empty()`                    ⟹ `range.is_empty()`
- `a.is_subrange(b)`                    ⟹ `b.contains_range(a)`
- `a.intersection(b)`                   ⟹ `a.intersect(b)`
- `a.extend_to(b)`                      ⟹ `a.cover(b)`
- `range.contains(offset)`              ⟹ `range.contains(point)`
- `range.contains_inclusive(offset)`    ⟹ `range.contains_inclusive(point)`
