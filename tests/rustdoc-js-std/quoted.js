// make sure quoted search works both for items and and without generics
// ignore-order

const FILTER_CRATE = 'std';

const EXPECTED = {
    'query': '"result"',
    'others': [
        { 'path': 'std', 'name': 'result' },
        { 'path': 'std::result', 'name': 'Result' },
        { 'path': 'std::fmt', 'name': 'Result' },
    ],
    'in_args': [
        { 'path': 'std::result::Result', 'name': 'branch' },
        { 'path': 'std::result::Result', 'name': 'ok' },
        { 'path': 'std::result::Result', 'name': 'unwrap' },
    ],
    'returned': [
        { 'path': 'std::bool', 'name': 'try_into' },
    ],
};
