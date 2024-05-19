// ignore-order

const EXPECTED = {
    'query': 'fn',
    'others': [
        { 'path': 'std', 'name': 'fn', ty: 1 }, // 1 is for primitive types
        { 'path': 'std', 'name': 'fn', ty: 0 }, // 0 is for keywords
    ],
};
