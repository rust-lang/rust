// ignore-order

const QUERY = 'fn';

const EXPECTED = {
    'others': [
        { 'path': 'std', 'name': 'fn', ty: 15 }, // 15 is for primitive types
        { 'path': 'std', 'name': 'fn', ty: 21 }, // 21 is for keywords
    ],
};
