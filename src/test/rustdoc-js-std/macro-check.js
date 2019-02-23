// ignore-order

const QUERY = 'panic';

const EXPECTED = {
    'others': [
        { 'path': 'std', 'name': 'panic', ty: 14 }, // 15 is for macros
        { 'path': 'std', 'name': 'panic', ty: 0 }, // 0 is for modules
    ],
};
