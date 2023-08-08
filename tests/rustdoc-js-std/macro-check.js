// ignore-order

const EXPECTED = {
    'query': 'panic',
    'others': [
        { 'path': 'std', 'name': 'panic', ty: 14 }, // 15 is for macros
        { 'path': 'std', 'name': 'panic', ty: 0 }, // 0 is for modules
    ],
};
